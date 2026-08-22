import os
import sys
from unittest.mock import MagicMock, AsyncMock
import pytest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from anthropic.lib.tools._skills import download_session_skills

@pytest.mark.asyncio
async def test_download_session_skills_multiagent():
    """Verify that skills are collected from both coordinator and roster agents."""
    client = AsyncMock()
    
    # Mock skill resolution and download to do nothing
    with patch("anthropic.lib.tools._skills._resolve_skill_version", new_callable=AsyncMock) as mock_resolve, \
         patch("anthropic.lib.tools._skills._download_and_extract", new_callable=AsyncMock) as mock_download:
        
        mock_resolve.return_value = "123456789"
        client.beta.skills.versions.retrieve = AsyncMock()
        client.beta.skills.versions.retrieve.return_value = MagicMock(name="test-skill")

        # Coordinator skills
        skill1 = MagicMock(skill_id="skill1", version="latest")
        
        # Roster agent skills
        skill2 = MagicMock(skill_id="skill2", version="latest")
        agent_entry = MagicMock()
        agent_entry.skills = [skill2]
        
        multiagent = MagicMock()
        multiagent.agents = [agent_entry]
        
        session = MagicMock()
        session.agent.skills = [skill1]
        session.agent.multiagent = multiagent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            await download_session_skills(client, workdir=tmpdir, session=session)
            
            # Should have called download for both skills
            assert mock_download.call_count == 2
            # args are (client, skill_id, version_id, dest)
            call_ids = [args[1] for args, _ in mock_download.call_args_list]
            assert "skill1" in call_ids
            assert "skill2" in call_ids

@pytest.mark.asyncio
async def test_download_session_skills_deduplication():
    """Verify that duplicate skills across agents are only downloaded once."""
    client = AsyncMock()
    
    with patch("anthropic.lib.tools._skills._resolve_skill_version", new_callable=AsyncMock) as mock_resolve, \
         patch("anthropic.lib.tools._skills._download_and_extract", new_callable=AsyncMock) as mock_download:
        
        mock_resolve.return_value = "123456789"
        client.beta.skills.versions.retrieve = AsyncMock()
        client.beta.skills.versions.retrieve.return_value = MagicMock(name="test-skill")

        skill1 = MagicMock(skill_id="shared-skill", version="latest")
        
        agent_entry = MagicMock()
        agent_entry.skills = [skill1]
        
        multiagent = MagicMock()
        multiagent.agents = [agent_entry]
        
        session = MagicMock()
        session.agent.skills = [skill1]
        session.agent.multiagent = multiagent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            await download_session_skills(client, workdir=tmpdir, session=session)
            
            # Should have called download only once
            assert mock_download.call_count == 1

if __name__ == "__main__":
    import asyncio
    import tempfile
    from unittest.mock import patch
    
    async def run_tests():
        await test_download_session_skills_multiagent()
        await test_download_session_skills_deduplication()
        print("Tests PASSED")
        
    asyncio.run(run_tests())
